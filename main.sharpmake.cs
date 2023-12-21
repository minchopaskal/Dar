using System.Collections.Generic;
using System.IO;
using Sharpmake;

static class Settings {
    public static Target GetTarget() {
        return new Target(
            Platform.win64,
            DevEnv.vs2022,
            Optimization.Debug | Optimization.Release);
    }

    public static Strings CommonLibraryFiles() {
        string[] static_libs = {"d3d12.lib", "dxcompiler.lib", "dxgi.lib", "dxguid.lib", "glfw3.lib" };
        return new Strings(static_libs);
    }

    public static void AddDxc(Project.Configuration conf) {
        conf.TargetCopyFiles.Add(@"[project.SharpmakeCsPath]/third_party/dxc_1.7/dxcompiler.dll");
        conf.TargetCopyFiles.Add(@"[project.SharpmakeCsPath]/third_party/dxc_1.7/dxil.dll");
    }

    public static void CopyCommonDLLs(Project.Configuration conf) {
        Settings.AddDxc(conf);
        conf.TargetCopyFilesToSubDirectory.Add(new KeyValuePair<string, string>(
              @"[project.SharpmakeCsPath]/lib/D3D12AgilitySDK/D3D12Core.dll", "D3D12")
        );
        conf.TargetCopyFilesToSubDirectory.Add(new KeyValuePair<string, string>(
              @"[project.SharpmakeCsPath]/lib/nvtt30204.dll", "./")
        );
    }
}

abstract class BaseProject : Project
{
    public BaseProject()
    {
        AddTargets(Settings.GetTarget());
    }

    [Configure]
    public virtual void ConfigureAll(Project.Configuration conf, Target target)
    {
        conf.Name = @"[target.Optimization]";
        conf.ProjectPath = Path.Combine(@"[project.SharpmakeCsPath]/solution/", @"[project.Name]");
        conf.Options.Add(Options.Vc.Compiler.CppLanguageStandard.CPP20);

        conf.LibraryPaths.Add(@"[project.SharpmakeCsPath]/lib");

        conf.IncludePaths.Add(@"[project.SharpmakeCsPath]");
        conf.IncludePaths.Add(@"[project.SharpmakeCsPath]/dar");
        conf.IncludePaths.Add(@"[project.SharpmakeCsPath]/dar/graphics");
        conf.IncludePaths.Add(@"[project.SharpmakeCsPath]/third_party");
        conf.IncludePaths.Add(@"[project.SharpmakeCsPath]/third_party/imgui");
        conf.IncludePaths.Add(@"[project.SharpmakeCsPath]/third_party/dxc_1.7");
        conf.IncludePaths.Add(@"[project.SharpmakeCsPath]/third_party/optick/src");

        conf.Defines.Add("_CRT_SECURE_NO_WARNINGS");
        string[] ignored_warnings = { "4201", "4996", "4099" };
        conf.Options.Add(new Sharpmake.Options.Vc.Compiler.DisableSpecificWarnings(ignored_warnings));

        if (target.Optimization == Optimization.Debug) {
            conf.Options.Add(Options.Vc.Compiler.RuntimeLibrary.MultiThreadedDebugDLL);
        } else {
            conf.Options.Add(Options.Vc.Compiler.RuntimeLibrary.MultiThreadedDLL);
        }

        conf.Options.Add(Options.Vc.General.WindowsTargetPlatformVersion.Latest);
    }

    [Configure(Optimization.Debug)]
    public virtual void ConfigureDebug(Project.Configuration conf, Target target) {
        conf.Defines.Add("DAR_DEBUG");
        conf.Defines.Add("DAR_PROFILE");
    }

    [Configure(Optimization.Release)]
    public virtual void ConfigureRelease(Project.Configuration conf, Target target) {
        conf.Defines.Add("DAR_NDEBUG");
    }
}

[Generate]
class ResourceManagerLib : BaseProject
{
    public ResourceManagerLib()
    {
        Name = "ResourceManagerLib";
        SourceRootPath = @"[project.SharpmakeCsPath]/reslib";
    }

    public override void ConfigureAll(Project.Configuration conf, Target target)
    {
        base.ConfigureAll(conf, target);
        conf.Output = Configuration.OutputType.Lib;
        conf.LibraryFiles.Add("nvtt.lib");
    }
}

[Generate]
class ResourceCompiler : BaseProject {
    public ResourceCompiler()
    {
        Name = "ResourceCompiler";
        SourceRootPath = @"[project.SharpmakeCsPath]/tools/resourcecompiler";
    }

    public override void ConfigureAll(Project.Configuration conf, Target target)
    {
        base.ConfigureAll(conf, target);
        Settings.AddDxc(conf);
        conf.ProjectPath = Path.Combine(@"[project.SharpmakeCsPath]/solution/tools/", @"[project.Name]");

        conf.TargetCopyFilesToSubDirectory.Add(new KeyValuePair<string, string>(
              @"[project.SharpmakeCsPath]/lib/nvtt30204.dll", "./")
        );

        conf.AddPrivateDependency<ResourceManagerLib>(target);
        conf.AddPrivateDependency<DarLibrary>(target);
    }
}

[Generate]
class ImGuiProject : BaseProject {
    public ImGuiProject()
    {
        Name = "ImGui";
        SourceRootPath = @"[project.SharpmakeCsPath]/third_party/imgui";
        
        SourceFilesExcludeRegex = new Strings(".*imgui_(?!draw|tables|widgets|impl_glfw|impl_dx12).*|examples.*");
    }

    public override void ConfigureAll(Project.Configuration conf, Target target)
    {
        base.ConfigureAll(conf, target);

        conf.Options.Add(Sharpmake.Options.Vc.General.WarningLevel.Level0);

        conf.LibraryFiles.Add("glfw3.lib");

        conf.Output = Configuration.OutputType.Lib;
   }
}

[Generate]
class OptickProject : BaseProject {
    public OptickProject()
    {
        Name = "Optick";
        SourceRootPath = @"[project.SharpmakeCsPath]/third_party/optick/src";

        AddTargets(Settings.GetTarget());
    }

    public override void ConfigureAll(Project.Configuration conf, Target target)
    {
        base.ConfigureAll(conf, target);

        conf.LibraryFiles.Add("dxgi.lib");
        conf.Defines.Add("OPTICK_ENABLE_GPU_VULKAN=0");
        conf.Output = Configuration.OutputType.Lib;
    }
}

[Generate]
class DarLibrary : BaseProject
{
    public DarLibrary()
    {
        Name = "Dar";
        SourceRootPath = @"[project.SharpmakeCsPath]/dar";
    }

    public override void ConfigureAll(Project.Configuration conf, Target target)
    {
        base.ConfigureAll(conf, target);
        foreach (string lib in Settings.CommonLibraryFiles()) {
            conf.LibraryFiles.Add(lib);
        }
        conf.Output = Configuration.OutputType.Lib;
        conf.AddPublicDependency<ResourceManagerLib>(target);
        conf.AddPublicDependency<ImGuiProject>(target);
        conf.AddPublicDependency<OptickProject>(target);
    }
}

[Generate]
class MikkTSpaceLib : BaseProject {
    public MikkTSpaceLib()
    {
        AddTargets(Settings.GetTarget());

        Name = "MikkTSpaceLib";
        SourceRootPath = @"[project.SharpmakeCsPath]/third_party/MikkTSpace";
    }

    public override void ConfigureAll(Project.Configuration conf, Target target)
    {
        base.ConfigureAll(conf, target);
        conf.Options.Add(new Sharpmake.Options.Vc.Compiler.DisableSpecificWarnings("4456", "4201"));
        conf.Output = Configuration.OutputType.Lib;
    }
}

[Generate]
class HelloTriangleProject : BaseProject
{
    public HelloTriangleProject() {
        Name = "HelloTriangle";
        SourceRootPath = @"[project.SharpmakeCsPath]/examples/hello_triangle";
        
        SourceFilesExtensions.Add("bat");

        ResourceFiles.Add(@"[project.SharpmakeCsPath]/third_party/res/shaders.shlib");
        
        ResourceFilesExtensions.Add("hlsl");
        ResourceFilesExtensions.Add("hlsli");
    }

    public override void ConfigureAll(Project.Configuration conf, Target target)
    {
        base.ConfigureAll(conf, target);
        foreach (string lib in Settings.CommonLibraryFiles()) {
            conf.LibraryFiles.Add(lib);
        }
        
        conf.TargetCopyFilesToSubDirectory.Add(new KeyValuePair<string, string>(
            @"[project.SourceRootPath]/res/shaders.shlib", "res/shaders")
        );
        conf.IncludePaths.Add(@"[project.SourceRootPath]/include");
        conf.IncludePaths.Add(@"[project.SourceRootPath]/res/shaders");

        Settings.CopyCommonDLLs(conf);

        conf.AddPrivateDependency<DarLibrary>(target);
        conf.AddPrivateDependency<ResourceManagerLib>(target);
        
        conf.VcxprojUserFile = new Project.Configuration.VcxprojUserFileSettings();
        conf.VcxprojUserFile.LocalDebuggerWorkingDirectory = "$(TargetDir)";
    }
}

[Generate]
class ShaderPlaythingProject : BaseProject
{
    public ShaderPlaythingProject() {
        Name = "ShaderPlaything";
        SourceRootPath = @"[project.SharpmakeCsPath]/examples/shader_plaything";

        SourceFiles.Add(@"[project.SharpmakeCsPath]/third_party/imguifiledialog/ImGuiFileDialog.cpp");
        SourceFiles.Add(@"[project.SharpmakeCsPath]/third_party/ImGuiColorTextEdit/TextEditor.cpp");

        SourceFilesExtensions.Add("bat");

        ResourceFiles.Add(@"[project.SharpmakeCsPath]/third_party/res/shaders.shlib");
        
        ResourceFilesExtensions.Add("hlsl");
        ResourceFilesExtensions.Add("hlsli");
    }

    public override void ConfigureAll(Project.Configuration conf, Target target)
    {
        base.ConfigureAll(conf, target);
        foreach (string lib in Settings.CommonLibraryFiles()) {
            conf.LibraryFiles.Add(lib);
        }
        
        conf.TargetCopyFilesToSubDirectory.Add(new KeyValuePair<string, string>(
            @"[project.SourceRootPath]/res/shaders/screen_quad.hlsli", "res/shaders")
        );
        conf.TargetCopyFilesToSubDirectory.Add(new KeyValuePair<string, string>(
            @"[project.SourceRootPath]/res/shaders/common.hlsli", "res/shaders")
        );

        conf.TargetCopyFilesToSubDirectory.Add(new KeyValuePair<string, string>(
            @"[project.SourceRootPath]/res/shaders.shlib", "res/shaders")
        );
        conf.IncludePaths.Add(@"[project.SourceRootPath]/include");
        conf.IncludePaths.Add(@"[project.SourceRootPath]/res/shaders");

        Settings.CopyCommonDLLs(conf);

        conf.AddPrivateDependency<DarLibrary>(target);
        conf.AddPrivateDependency<ResourceManagerLib>(target);
        
        conf.VcxprojUserFile = new Project.Configuration.VcxprojUserFileSettings();
        conf.VcxprojUserFile.LocalDebuggerWorkingDirectory = "$(TargetDir)";
    }
}

[Generate]
class SponzaProject : BaseProject
{
    public SponzaProject() {
        Name = "Sponza";
        SourceRootPath = @"[project.SharpmakeCsPath]/examples/sponza";
        
        SourceFilesExtensions.Add("bat");

        ResourceFiles.Add(@"[project.SharpmakeCsPath]/third_party/res/shaders.shlib");
        ResourceFiles.Add(@"[project.SharpmakeCsPath]/third_party/res/textures.txlib");
        
        ResourceFilesExtensions.Add("hlsl");
        ResourceFilesExtensions.Add("hlsli");
        ResourceFilesExtensions.Add("bin");
        ResourceFilesExtensions.Add("gltf");
        ResourceFilesExtensions.Add("json");
    }

    public override void ConfigureAll(Project.Configuration conf, Target target)
    {
        base.ConfigureAll(conf, target);
        foreach (string lib in Settings.CommonLibraryFiles()) {
            conf.LibraryFiles.Add(lib);
        }
        
        conf.TargetCopyFilesToSubDirectory.Add(new KeyValuePair<string, string>(
            @"[project.SourceRootPath]/res/shaders.shlib", "res/shaders")
        );
        conf.TargetCopyFilesToSubDirectory.Add(new KeyValuePair<string, string>(
            @"[project.SourceRootPath]/res/textures.txlib", "res/textures")
        );

        conf.IncludePaths.Add(@"[project.SourceRootPath]/include");
        conf.IncludePaths.Add(@"[project.SourceRootPath]/res/shaders");
        conf.IncludePaths.Add(@"[project.SharpmakeCsPath]/third_party/assimp/include");

        Settings.CopyCommonDLLs(conf);

        // Hacky but what can you do?
        conf.TargetCopyFilesToSubDirectory.Add(new KeyValuePair<string, string>(
            @"[project.SourceRootPath]/res/scenes/Sponza/Sponza.gltf", "res/scenes/Sponza"));
        conf.TargetCopyFilesToSubDirectory.Add(new KeyValuePair<string, string>(
            @"[project.SourceRootPath]/res/scenes/Sponza/Sponza.bin", "res/scenes/Sponza")
        );
        conf.TargetCopyFilesToSubDirectory.Add(new KeyValuePair<string, string>(
            @"[project.SourceRootPath]/res/scenes/sponza.json", "res/scenes/"));

        conf.AddPrivateDependency<DarLibrary>(target);
        conf.AddPrivateDependency<MikkTSpaceLib>(target);
        conf.AddPrivateDependency<ResourceManagerLib>(target);
        
        conf.VcxprojUserFile = new Project.Configuration.VcxprojUserFileSettings();
        // TODO: This is hack, see how to call Sharpmake's path resolver
        conf.VcxprojUserFile.LocalDebuggerWorkingDirectory = "$(TargetDir)";

        conf.Options.Add(Options.Vc.Compiler.RTTI.Enable);

        if (target.Optimization == Optimization.Debug) {
            conf.LibraryFiles.Add("assimpd.lib");
        } else {
            conf.LibraryFiles.Add("assimp.lib");
        }
    }
}

[Generate]
public class DarProjectSolution : Solution
{
    public DarProjectSolution()
    {
        Name = "Dar";
        AddTargets(Settings.GetTarget());
    }

    [Configure]
    public void ConfigureAll(Solution.Configuration conf, Target target)
    {
        conf.Name = @"[target.Optimization]";
        conf.SolutionPath = @"[solution.SharpmakeCsPath]\solution";
        conf.AddProject<SponzaProject>(target);
        conf.AddProject<HelloTriangleProject>(target);
        conf.AddProject<ShaderPlaythingProject>(target);
    }
}

[Generate]
public class ToolsSolution : Solution
{
    public ToolsSolution()
    {
        Name = "ToolsSolution";
        AddTargets(Settings.GetTarget());
    }

    [Configure]
    public void ConfigureAll(Solution.Configuration conf, Target target)
    {
        conf.Name = @"[target.Optimization]";
        conf.SolutionPath = @"[solution.SharpmakeCsPath]/solution/tools";
        conf.AddProject<ResourceCompiler>(target);
    }
}

public static class Main
{
    [Sharpmake.Main]
    public static void SharpmakeMain(Sharpmake.Arguments args) {
        args.Generate<DarProjectSolution>();
        args.Generate<ToolsSolution>();
    }
}
